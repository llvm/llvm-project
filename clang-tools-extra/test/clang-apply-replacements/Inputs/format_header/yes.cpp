#include <string>
// CHECK: #include "bar.h"
// CHECK-NEXT: #include <memory>
// CHECK-NEXT: #include <string>
#include <memory>
#include "foo.h"
#include "bar.h"

void foo() {
}
