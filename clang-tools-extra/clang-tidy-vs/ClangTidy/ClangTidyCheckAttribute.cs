using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LLVM.ClangTidy
{
    public class ClangTidyCheckAttribute : Attribute
    {
        private string CheckName_;
        public ClangTidyCheckAttribute(string CheckName)
        {
            this.CheckName_ = CheckName;
        }

        public string CheckName
        {
            get { return CheckName_; }
        }
    }
}
