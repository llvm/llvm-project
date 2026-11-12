#ifndef _GLIBCXX_STRING
#define _GLIBCXX_STRING 1

template<typename T>
struct basic_string {
  static T _S_empty_rep_storage[];
};

template<typename T>
T basic_string<T>::_S_empty_rep_storage[sizeof(T)];

#endif
