namespace absl {
namespace base_internal {
void InternalFunction() {}
} // namespace base_internal 
} //namespace absl
void DirectAccess2() { absl::base_internal::InternalFunction(); }
