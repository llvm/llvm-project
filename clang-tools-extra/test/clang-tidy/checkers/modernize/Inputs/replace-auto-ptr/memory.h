#ifndef INPUTS_MEMORY_H
#define INPUTS_MEMORY_H

namespace std {

inline namespace _1 {

template <class Y> struct auto_ptr_ref {
  Y *y_;
};

template <class X> class auto_ptr {
public:
  typedef X element_type;
  explicit auto_ptr(X *p = 0) throw() {}
  auto_ptr(auto_ptr &) throw() {}
  template <class Y> auto_ptr(auto_ptr<Y> &) throw() {}
  auto_ptr &operator=(auto_ptr &) throw() { return *this; }
  template <class Y> auto_ptr &operator=(auto_ptr<Y> &) throw() {
    return *this;
  }
  auto_ptr &operator=(auto_ptr_ref<X> r) throw() { return *this; }
  ~auto_ptr() throw() {}
  auto_ptr(auto_ptr_ref<X> r) throw() : x_(r.y_) {}
  template <class Y> operator auto_ptr_ref<Y>() throw() {
    auto_ptr_ref<Y> r;
    r.y_ = x_;
    return r;
  }
  template <class Y> operator auto_ptr<Y>() throw() { return auto_ptr<Y>(x_); }

private:
  X *x_;
};

template <> class auto_ptr<void> {
public:
  typedef void element_type;
};

} // namespace _1

} // end namespace std

#endif // INPUTS_MEMORY_H
