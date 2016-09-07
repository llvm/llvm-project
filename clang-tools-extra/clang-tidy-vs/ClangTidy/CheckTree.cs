using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;

namespace LLVM.ClangTidy
{
    /// <summary>
    /// CheckTree is used to group checks into categories and subcategories.  For
    /// example, given the following list of checks:
    /// 
    ///   llvm-include-order
    ///   llvm-namespace-comment
    ///   llvm-twine-local
    ///   llvm-header-guard
    ///   google-runtime-member-string-references
    ///   google-runtime-int
    ///   google-readability-namespace-comments
    ///   
    /// the corresponding CheckTree would look like this:
    /// 
    ///   llvm
    ///     include-order
    ///     namespace-comment
    ///     twine-local
    ///     header-guard
    ///   google
    ///     runtime
    ///       member-string-references
    ///       int
    ///     readability
    ///       namespace-comments
    ///       redundant-smartptr-get
    ///       
    /// This is useful when serializing a set of options out to a .clang-tidy file,
    /// because we need to decide the most efficient way to serialize the sequence
    /// of check commands, when to use wildcards, etc.  For example, if everything
    /// under google is inherited, we can simply leave that entry out entirely from
    /// the .clang-tidy file.  On the other hand, if anything is inherited, we *must
    /// not* add or remove google-* by wildcard because that, by definition, means
    /// the property is no longer inherited.  When we can categorize the checks into
    /// groups and subgroups like this, it is possible to efficiently serialize to
    /// a minimal representative .clang-tidy file.
    /// </summary>

    public abstract class CheckTreeNode
    {
        private string Name_;
        private CheckTreeNode Parent_;

        protected CheckTreeNode(string Name, CheckTreeNode Parent)
        {
            Name_ = Name;
            Parent_ = Parent;
        }

        public string Path
        {
            get
            {
                if (Parent_ == null)
                    return null;
                string ParentPath = Parent_.Path;
                if (ParentPath == null)
                    return Name_;
                return ParentPath + "-" + Name_;
            }
        }

        public string Name
        {
            get
            {
                return Name_;
            }
        }


        public abstract int CountChecks { get; }
        public abstract int CountExplicitlyDisabledChecks { get; }
        public abstract int CountExplicitlyEnabledChecks { get; }
        public abstract int CountInheritedChecks { get; }
    }

    public class CheckTree : CheckTreeNode
    {
        private Dictionary<string, CheckTreeNode> Children_ = new Dictionary<string, CheckTreeNode>();
        public CheckTree()
            : base(null, null)
        {

        }

        private CheckTree(string Name, CheckTree Parent)
            : base(Name, Parent)
        {
        }

        private void AddLeaf(string Name, DynamicPropertyDescriptor<bool> Property)
        {
            Children_[Name] = new CheckLeaf(Name, this, Property);
        }

        private CheckTree AddOrCreateSubgroup(string Name)
        {
            CheckTreeNode Subgroup = null;
            if (Children_.TryGetValue(Name, out Subgroup))
            {
                System.Diagnostics.Debug.Assert(Subgroup is CheckTree);
                return (CheckTree)Subgroup;
            }

            CheckTree SG = new CheckTree(Name, this);
            Children_[Name] = SG;
            return SG;
        }

        public static CheckTree Build(ClangTidyProperties Config)
        {
            // Since some check names contain dashes in them, it doesn't make sense to
            // simply split all check names by dash and construct a huge tree.  For
            // example, in the check called google-runtime-member-string-references,
            // we don't need each of those to be a different subgroup.  So instead we
            // explicitly specify the common breaking points at which a user might want
            // to use a -* and everything else falls as a leaf under one of these
            // categories.
            // FIXME: This should be configurable without recompilation
            CheckTree Root = new CheckTree();
            string[][] Groups = new string[][] {
                new string[] {"boost"},
                new string[] {"cert"},
                new string[] {"clang", "diagnostic"},
                new string[] {"cppcoreguidelines", "interfaces"},
                new string[] {"cppcoreguidelines", "pro", "bounds"},
                new string[] {"cppcoreguidelines", "pro", "type"},
                new string[] {"google", "build"},
                new string[] {"google", "readability"},
                new string[] {"google", "runtime"},
                new string[] {"llvm"},
                new string[] {"misc"},
            };

            foreach (string[] Group in Groups)
            {
                CheckTree Subgroup = Root;
                foreach (string Component in Group)
                    Subgroup = Subgroup.AddOrCreateSubgroup(Component);
            }

            var Props = Config.GetProperties()
                              .Cast<PropertyDescriptor>()
                              .OfType<DynamicPropertyDescriptor<bool>>()
                              .Where(x => x.Attributes.OfType<ClangTidyCheckAttribute>().Count() > 0)
                              .Select(x => new KeyValuePair<DynamicPropertyDescriptor<bool>, string>(
                                            x, x.Attributes.OfType<ClangTidyCheckAttribute>().First().CheckName));
            var PropArray = Props.ToArray();
            foreach (var CheckInfo in PropArray)
            {
                string LeafName = null;
                CheckTree Tree = Root.LocateCheckLeafGroup(CheckInfo.Value, out LeafName);
                Tree.AddLeaf(LeafName, CheckInfo.Key);
            }
            return Root;
        }

        private CheckTree LocateCheckLeafGroup(string Check, out string LeafName)
        {
            string[] Components = Check.Split('-');
            string FirstComponent = Components.FirstOrDefault();
            if (FirstComponent == null)
            {
                LeafName = Check;
                return this;
            }

            CheckTreeNode Subgroup = null;
            if (!Children_.TryGetValue(FirstComponent, out Subgroup))
            {
                LeafName = Check;
                return this;
            }
            System.Diagnostics.Debug.Assert(Subgroup is CheckTree);
            CheckTree Child = (CheckTree)Subgroup;
            string ChildName = Check.Substring(FirstComponent.Length + 1);
            return Child.LocateCheckLeafGroup(ChildName, out LeafName);
        }

        public override int CountChecks
        {
            get
            {
                return Children_.Aggregate(0, (X, V) => { return X + V.Value.CountChecks; });
            }
        }

        public override int CountExplicitlyDisabledChecks
        {
            get
            {
                return Children_.Aggregate(0, (X, V) => { return X + V.Value.CountExplicitlyDisabledChecks; });
            }
        }

        public override int CountExplicitlyEnabledChecks
        {
            get
            {
                return Children_.Aggregate(0, (X, V) => { return X + V.Value.CountExplicitlyEnabledChecks; });
            }
        }
        public override int CountInheritedChecks
        {
            get
            {
                return Children_.Aggregate(0, (X, V) => { return X + V.Value.CountInheritedChecks; });
            }
        }

        public IDictionary<string, CheckTreeNode> Children
        {
            get { return Children_; }
        }
    }

    public class CheckLeaf : CheckTreeNode
    {
        private DynamicPropertyDescriptor<bool> Property_;

        public CheckLeaf(string Name, CheckTree Parent, DynamicPropertyDescriptor<bool> Property)
            : base(Name, Parent)
        {
            Property_ = Property;
        }

        public override int CountChecks
        {
            get
            {
                return 1;
            }
        }

        public override int CountExplicitlyDisabledChecks
        {
            get
            {
                if (Property_.IsInheriting)
                    return 0;
                return (bool)Property_.GetValue(null) ? 0 : 1;
            }
        }

        public override int CountExplicitlyEnabledChecks
        {
            get
            {
                if (Property_.IsInheriting)
                    return 0;
                return (bool)Property_.GetValue(null) ? 1 : 0;
            }
        }

        public override int CountInheritedChecks
        {
            get
            {
                return (Property_.IsInheriting) ? 1 : 0;
            }
        }

    }
}
