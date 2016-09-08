using System;

namespace LLVM.ClangTidy
{
    static class GuidList
    {
        public const string guidClangTidyPkgString = "AE4956BE-3DB8-430E-BBAB-7E2E9A014E9C";
        public const string guidClangTidyCmdSetString = "9E0F0493-6493-46DE-AEE1-ACD8F60F265E";

        public static readonly Guid guidClangTidyCmdSet = new Guid(guidClangTidyCmdSetString);
    };
}
