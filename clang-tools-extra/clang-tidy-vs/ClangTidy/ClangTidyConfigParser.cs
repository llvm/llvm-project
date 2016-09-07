using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace LLVM.ClangTidy
{
    static class ClangTidyConfigParser
    {
        public class CheckOption
        {
            [YamlAlias("key")]
            public string Key { get; set; }

            [YamlAlias("value")]
            public string Value { get; set; }
        }
        public class ClangTidyYaml
        {
            [YamlAlias("Checks")]
            public string Checks { get; set; }

            [YamlAlias("CheckOptions")]
            public List<CheckOption> CheckOptions { get; set; }
        }

        public static List<KeyValuePair<string, ClangTidyProperties>> ParseConfigurationChain(string ClangTidyFile)
        {
            List<KeyValuePair<string, ClangTidyProperties>> Result = new List<KeyValuePair<string, ClangTidyProperties>>();
            Result.Add(new KeyValuePair<string, ClangTidyProperties>(null, ClangTidyProperties.RootProperties));

            foreach (string P in Utility.SplitPath(ClangTidyFile).Reverse())
            {
                if (!Utility.HasClangTidyFile(P))
                    continue;

                string ConfigFile = Path.Combine(P, ".clang-tidy");

                using (StreamReader Reader = new StreamReader(ConfigFile))
                {
                    Deserializer D = new Deserializer(namingConvention: new PascalCaseNamingConvention());
                    ClangTidyYaml Y = D.Deserialize<ClangTidyYaml>(Reader);
                    ClangTidyProperties Parent = Result[Result.Count - 1].Value;
                    ClangTidyProperties NewProps = new ClangTidyProperties(Parent);
                    SetPropertiesFromYaml(Y, NewProps);
                    Result.Add(new KeyValuePair<string, ClangTidyProperties>(P, NewProps));
                }
            }
            return Result;
        }

        enum TreeLevelOp
        {
            Enable,
            Disable,
            Inherit
        }

        public static void SerializeClangTidyFile(ClangTidyProperties Props, string ClangTidyFilePath)
        {
            List<string> CommandList = new List<string>();
            SerializeCheckTree(CommandList, Props.GetCheckTree(), TreeLevelOp.Inherit);

            CommandList.Sort((x, y) =>
            {
                bool LeftSub = x.StartsWith("-");
                bool RightSub = y.StartsWith("-");
                if (LeftSub && !RightSub)
                    return -1;
                if (RightSub && !LeftSub)
                    return 1;
                return StringComparer.CurrentCulture.Compare(x, y);
            });

            string ConfigFile = Path.Combine(ClangTidyFilePath, ".clang-tidy");
            using (StreamWriter Writer = new StreamWriter(ConfigFile))
            {
                Serializer S = new Serializer(namingConvention: new PascalCaseNamingConvention());
                ClangTidyYaml Yaml = new ClangTidyYaml();
                Yaml.Checks = String.Join(",", CommandList.ToArray());
                S.Serialize(Writer, Yaml);
            }
        }

        /// <summary>
        /// Convert the given check tree into serialized list of commands that can be written to
        /// the Yaml.  The goal here is to determine the minimal sequence of check commands that
        /// will produce the exact configuration displayed in the UI.  This is complicated by the
        /// fact that an inherited True is not the same as an explicitly specified True.  If the
        /// user has chosen to inherit a setting in a .clang-tidy file, then changing it in the
        /// parent should show the reflected changes in the current file as well.  So we cannot
        /// simply -* everything and then add in the checks we need, because -* immediately marks
        /// every single check as explicitly false, thus disabling inheritance.
        /// </summary>
        /// <param name="CommandList">State passed through this recursive algorithm representing
        /// the sequence of commands we have determined so far.
        /// </param>
        /// <param name="Tree">The check tree to serialize.  This is the parameter that will be
        /// recursed on as successive subtrees get serialized to `CommandList`.
        /// </param>
        /// <param name="CurrentOp">The current state of the subtree.  For example, if the
        /// algorithm decides to -* an entire subtree and then add back one single check,
        /// after adding a -subtree-* command to CommandList, it would pass in a value of
        /// CurrentOp=TreeLevelOp.Disable when it recurses down.  This allows deeper iterations
        /// of the algorithm to know what kind of command (if any) needs to be added to CommandList
        /// in order to put a particular check into a particular state.
        /// </param>
        private static void SerializeCheckTree(List<string> CommandList, CheckTree Tree, TreeLevelOp CurrentOp)
        {
            int NumChecks = Tree.CountChecks;
            int NumDisabled = Tree.CountExplicitlyDisabledChecks;
            int NumEnabled = Tree.CountExplicitlyEnabledChecks;
            int NumInherited = Tree.CountInheritedChecks;

            if (NumChecks == 0)
                return;

            if (NumInherited > 0)
                System.Diagnostics.Debug.Assert(CurrentOp == TreeLevelOp.Inherit);

            // If this entire tree is inherited, just exit, nothing about this needs to
            // go in the clang-tidy file.
            if (NumInherited == NumChecks)
                return;

            TreeLevelOp NewOp = CurrentOp;
            // If there are no inherited properties in this subtree, decide whether to
            // explicitly enable or disable this subtree.  Decide by looking at whether
            // there is a larger proportion of disabled or enabled descendants.  If
            // there are more disabled items in this subtree for example, disabling the
            // subtree will lead to a smaller configuration file.
            if (NumInherited == 0)
            {
                if (NumDisabled >= NumEnabled)
                    NewOp = TreeLevelOp.Disable;
                else
                    NewOp = TreeLevelOp.Enable;
            }

            if (NewOp == TreeLevelOp.Disable)
            {
                // Only add an explicit disable command if the tree was not already disabled
                // to begin with.
                if (CurrentOp != TreeLevelOp.Disable)
                {
                    string WildcardPath = "*";
                    if (Tree.Path != null)
                        WildcardPath = Tree.Path + "-" + WildcardPath;
                    CommandList.Add("-" + WildcardPath);
                }
                // If the entire subtree was disabled, there's no point descending.
                if (NumDisabled == NumChecks)
                    return;
            }
            else if (NewOp == TreeLevelOp.Enable)
            {
                // Only add an explicit enable command if the tree was not already enabled
                // to begin with.  Note that if we're at the root, all checks are already
                // enabled by default, so there's no need to explicitly include *
                if (CurrentOp != TreeLevelOp.Enable && Tree.Path != null)
                {
                    string WildcardPath = Tree.Path + "-*";
                    CommandList.Add(WildcardPath);
                }
                // If the entire subtree was enabled, there's no point descending.
                if (NumEnabled == NumChecks)
                    return;
            }

            foreach (var Child in Tree.Children)
            {
                if (Child.Value is CheckLeaf)
                {
                    CheckLeaf Leaf = (CheckLeaf)Child.Value;
                    if (Leaf.CountExplicitlyEnabledChecks == 1 && NewOp != TreeLevelOp.Enable)
                        CommandList.Add(Leaf.Path);
                    else if (Leaf.CountExplicitlyDisabledChecks == 1 && NewOp != TreeLevelOp.Disable)
                        CommandList.Add("-" + Leaf.Path);
                    continue;
                }

                System.Diagnostics.Debug.Assert(Child.Value is CheckTree);
                CheckTree ChildTree = (CheckTree)Child.Value;
                SerializeCheckTree(CommandList, ChildTree, NewOp);
            }
        }

        private static void SetPropertiesFromYaml(ClangTidyYaml Yaml, ClangTidyProperties Props)
        {
            string[] CheckCommands = Yaml.Checks.Split(',');
            foreach (string Command in CheckCommands)
            {
                if (Command == null || Command.Length == 0)
                    continue;
                bool Add = true;
                string Pattern = Command;
                if (Pattern[0] == '-')
                {
                    Pattern = Pattern.Substring(1);
                    Add = false;
                }

                foreach (var Match in CheckDatabase.Checks.Where(x => Utility.MatchWildcardString(x.Name, Pattern)))
                {
                    Props.SetDynamicValue(Match.Name, Add);
                }
            }
        }
    }
}
