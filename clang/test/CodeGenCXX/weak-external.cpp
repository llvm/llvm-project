// RUN: %clang -fexceptions %s -S -emit-llvm -o - | FileCheck %s
// PR4262

// CHECK-NOT: _ZNSs12_S_constructIPKcEEPcT_S3_RKSaIcESt20forward_iterator_tag

// The "basic_string" extern template instantiation declaration is supposed to
// suppress the implicit instantiation of non-inline member functions. Make sure
// that we suppress the implicit instantiation of non-inline member functions
// defined out-of-line. That we aren't instantiating the basic_string
// constructor when we shouldn't be. Such an instantiation forces the implicit
// instantiation of _S_construct<const char*>. Since _S_construct is a member
// template, it's instantiation is *not* suppressed (despite being in
// basic_string<char>), so we would emit it as a weak definition.

#define _LIBCPP_EXCEPTION_ABI __attribute__ ((__visibility__("default")))
#define _LIBCPP_INLINE_VISIBILITY __attribute__ ((__visibility__("hidden"), __always_inline__))
#define _LIBCPP_VISIBLE __attribute__ ((__visibility__("default")))
#if (__has_feature(cxx_noexcept))
#  define _NOEXCEPT noexcept
#  define _NOEXCEPT_(x) noexcept(x)
#else
#  define _NOEXCEPT throw()
#  define _NOEXCEPT_(x)
#endif

namespace std  // purposefully not using versioning namespace
{

template<class charT> struct char_traits;
template<class T>     class allocator;
template <class _CharT,
          class _Traits = char_traits<_CharT>,
          class _Allocator = allocator<_CharT> >
    class _LIBCPP_VISIBLE basic_string;
typedef basic_string<char, char_traits<char>, allocator<char> > string;

class _LIBCPP_EXCEPTION_ABI exception
{
public:
    _LIBCPP_INLINE_VISIBILITY exception() _NOEXCEPT {}
    virtual ~exception() _NOEXCEPT;
    virtual const char* what() const _NOEXCEPT;
};

class _LIBCPP_EXCEPTION_ABI runtime_error
    : public exception
{
private:
    void* __imp_;
public:
    explicit runtime_error(const string&);
    explicit runtime_error(const char*);

    runtime_error(const runtime_error&) _NOEXCEPT;
    runtime_error& operator=(const runtime_error&) _NOEXCEPT;

    virtual ~runtime_error() _NOEXCEPT;

    virtual const char* what() const _NOEXCEPT;
};

}

void dummysymbol() {
  throw(std::runtime_error("string"));
}
