using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace LLVM.ClangTidy
{
    static class Utility
    {
        public static IEnumerable<string> SplitPath(string FileOrDir)
        {
            string P = Path.GetDirectoryName(FileOrDir);
            do
            {
                yield return P;
                P = Path.GetDirectoryName(P);
            } while (P != null);
        }

        public static bool HasClangTidyFile(string Folder)
        {
            string ClangTidy = Path.Combine(Folder, ".clang-tidy");
            return File.Exists(ClangTidy);
        }

        public static bool MatchWildcardString(string Value, string Pattern)
        {
            string RE = Regex.Escape(Pattern).Replace(@"\*", ".*");
            return Regex.IsMatch(Value, RE);
        }
    }
}
