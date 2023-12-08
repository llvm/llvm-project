#ifndef _GLIBCXX_STRING
#define _GLIBCXX_STRING
template<typename> struct basic_string {
  struct _Alloc_hider {} _M_dataplus;
  ~basic_string() { _Alloc_hider h; } 
};
extern template class basic_string<char>;
#endif
