#include "clang/Sema/SemaSummarizer.h"

namespace clang {
void SemaSummarizer::SummarizeFunctionBody(FunctionDecl *FD) const {
  FD->dump();
}

} // namespace clang