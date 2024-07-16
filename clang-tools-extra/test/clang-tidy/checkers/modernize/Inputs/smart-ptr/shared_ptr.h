namespace std {

template <typename type>
class __shared_ptr {
protected:
  __shared_ptr();
  __shared_ptr(type *ptr);
  ~__shared_ptr();
public:
  type &operator*() { return *ptr; }
  type *operator->() { return ptr; }
  type *get() const;
  type *release();
  void reset();
  void reset(type *pt);
  explicit operator bool() const;

private:
  type *ptr;
};

template <typename type>
class shared_ptr : public __shared_ptr<type> {
public:
  shared_ptr();
  shared_ptr(type *ptr);
  shared_ptr(const shared_ptr<type> &t);
  shared_ptr(shared_ptr<type> &&t);
  ~shared_ptr();
  shared_ptr &operator=(shared_ptr &&);
  template <typename T>
  shared_ptr &operator=(shared_ptr<T> &&);
};

}  // namespace std
