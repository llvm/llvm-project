namespace {
void func1() {
}
} // namespace

namespace test_namespace1 {
namespace {
void func2() {}
} // namespace
} // namespace test_namespace1

namespace test_namespace2 {
namespace {
namespace test_namespace3 {
void func3() {}
} // namespace test_namespace3
} // namespace
} // namespace test_namespace2

namespace {
namespace {
void func4() {
}
} // namespace
} // namespace
