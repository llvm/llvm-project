
namespace clang {
namespace diag {
enum {
#define DIAG(ENUM, FLAGS, DEFAULT_MAPPING, DESC, GROUP, SFINAE, NOWERROR,      \
             SHOWINSYSHEADER, SHOWINSYSMACRO, DEFERRABLE, CATEGORY)            \
  ENUM,
#define LEXSTART
#include "clang/Basic/DiagnosticLexKinds.inc"
#undef DIAG
  NUM_BUILTIN_LEX_DIAGNOSTICS
};

#define DIAG_ENUM(ENUM_NAME)                                                   \
  namespace ENUM_NAME {                                                        \
  enum {
#define DIAG_ENUM_ITEM(IDX, NAME) NAME = IDX,
#define DIAG_ENUM_END()                                                        \
  }                                                                            \
  ;                                                                            \
  }
#include "clang/Basic/DiagnosticLexEnums.inc"
#undef DIAG_ENUM_END
#undef DIAG_ENUM_ITEM
#undef DIAG_ENUM
} // end namespace diag

namespace diag_compat {
#define DIAG_COMPAT_IDS_BEGIN() enum {
#define DIAG_COMPAT_IDS_END()                                                  \
  }                                                                            \
  ;
#define DIAG_COMPAT_ID(IDX, NAME, ...) NAME = IDX,
#include "clang/Basic/DiagnosticLexCompatIDs.inc"
#undef DIAG_COMPAT_ID
#undef DIAG_COMPAT_IDS_BEGIN
#undef DIAG_COMPAT_IDS_END
} // end namespace diag_compat
} // end namespace clang
