template<typename T, T v> struct c {};
using d = c<bool, false>;
struct foo : public d {};
