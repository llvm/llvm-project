namespace std {

template <typename type>
class shared_ptr {
public:
  shared_ptr();
  shared_ptr(type *ptr);
  shared_ptr(const shared_ptr<type> &t) {}
  shared_ptr(shared_ptr<type> &&t) {}
  ~shared_ptr();
  type &operator*() { return *ptr; }
  type *operator->() { return ptr; }
  type *release();
  void reset();
  void reset(type *pt);
  shared_ptr &operator=(shared_ptr &&);
  template <typename T>
  shared_ptr &operator=(shared_ptr<T> &&);

private:
  type *ptr;
};

}  // namespace std
