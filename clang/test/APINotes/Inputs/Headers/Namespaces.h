namespace Namespace1 { namespace Nested1 {} }

namespace Namespace1 {
static int varInNamespace = 1;
struct char_box { char c; };
void funcInNamespace();

namespace Nested1 {
void funcInNestedNamespace(int i);
struct char_box {
  char c;
};
}

namespace Nested1 {
static int varInNestedNamespace = 1;
void funcInNestedNamespace(int i);

namespace Namespace1 {
struct char_box { char c; };
} // namespace Namespace1
} // namespace Nested1

namespace Nested2 {
static int varInNestedNamespace = 2;
} // namespace Nested2

namespace Nested1 { namespace Namespace1 {} }
} // namespace Namespace1

namespace Namespace1 {
typedef int my_typedef;
using my_using_decl = int;
}

inline namespace InlineNamespace1 {
static int varInInlineNamespace = 3;
void funcInInlineNamespace();
}
