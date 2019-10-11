namespace std {
struct string {
  string(const char *);
  ~string();
};
} // namespace std

namespace absl {
std::string StringsFunction(std::string s1) { return s1; }
class SomeContainer {};
namespace strings_internal {
void InternalFunction() {}
template <class P> P InternalTemplateFunction(P a) {}
} // namespace strings_internal

namespace container_internal {
struct InternalStruct {};
} // namespace container_internal
} // namespace absl

// should not trigger warnings because inside Abseil files
void DirectAcessInternal() {
  absl::strings_internal::InternalFunction();
  absl::strings_internal::InternalTemplateFunction<std::string>("a");
}

class FriendUsageInternal {
  friend struct absl::container_internal::InternalStruct;
};

namespace absl {
void OpeningNamespaceInternally() { strings_internal::InternalFunction(); }
} // namespace absl
