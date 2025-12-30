#include <string>
// CHECK: #include <string>
// CHECK-NEXT: #include <memory>
// CHECK-NEXT: #include "bar.h"
#include <memory>
#include "foo.h"
#include "bar.h"

void foo() {
}
