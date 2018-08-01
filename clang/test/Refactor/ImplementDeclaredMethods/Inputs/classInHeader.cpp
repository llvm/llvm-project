#include "classInHeader.h"

#ifndef NO_IMPL

#define PREFIX

#ifdef USE_NAMESPACE
#ifdef USE_NAMESPACE_PREFIX
#define PREFIX ns::ns2::
#else
#ifdef USE_NAMESPACE_USING
using namespace ns::ns2;
#else
namespace ns {
namespace ns2 {
#define CLOSE_NAMESPACES
#endif
#endif
#endif

void PREFIX ClassInHeader::implementedToo() {

}

void PREFIX ClassInHeader::implemented() {

}
// CHECK1: "{{.*}}classInHeader.cpp" "\n\nvoid ClassInHeader::pleaseImplement() { \n  <#code#>;\n}\n\nvoid ClassInHeader::pleaseImplementThisAsWell() { \n  <#code#>;\n}\n" [[@LINE-1]]:2
// CHECK1-NS-PREFIX: "{{.*}}classInHeader.cpp" "\n\nvoid ns::ns2::ClassInHeader::pleaseImplement() { \n  <#code#>;\n}\n\nvoid ns::ns2::ClassInHeader::pleaseImplementThisAsWell() { \n  <#code#>;\n}\n" [[@LINE-2]]:2

#ifdef CLOSE_NAMESPACES
}
}
#endif

#endif

namespace other {
#ifndef USE_NAMESPACE_USING
using namespace ns::ns2;
#else
}

void usingCanBeHidden() {
#ifndef USE_NAMESPACE_USING
using namespace ns::ns2;
#else
}

#ifdef USE_NAMESPACE_USING
using namespace ns::ns2;
#else
// We still want to insert 'using namespace ns::ns2' if the outer is already
// used.
using namespace ns;
#endif

using namespace other;

namespace ns {
namespace ns2 {
// Prefer to insert the methods at the end using 'using' instead of into a
// namespace.
}
}

// CHECK1-NO-IMPL-USING-NS-IN-RECORD: "{{.*}}classInHeader.cpp" "\nusing namespace ns::ns2;\n\nvoid OuterRecord::ClassInHeader::pleaseImplement() { \n  <#code#>;\n}\n\nvoid OuterRecord::ClassInHeader::pleaseImplementThisAsWell() { \n  <#code#>;\n}\n" [[@LINE+3]]:1
// CHECK1-NO-IMPL-USING-NS: "{{.*}}classInHeader.cpp" "\nusing namespace ns::ns2;\n\nvoid ClassInHeader::pleaseImplement() { \n  <#code#>;\n}\n\nvoid ClassInHeader::pleaseImplementThisAsWell() { \n  <#code#>;\n}\n" [[@LINE+2]]:1
// CHECK1-NO-IMPL: "{{.*}}classInHeader.cpp" "\n\nvoid ClassInHeader::pleaseImplement() { \n  <#code#>;\n}\n\nvoid ClassInHeader::pleaseImplementThisAsWell() { \n  <#code#>;\n}\n" [[@LINE+1]]:1 -> [[@LINE+1]]:1
