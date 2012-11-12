#pragma clang system_header

namespace std {
  template <class T1, class T2>
  struct pair {
    T1 first;
    T2 second;
    
    pair() : first(), second() {}
    pair(const T1 &a, const T2 &b) : first(a), second(b) {}
    
    template<class U1, class U2>
    pair(const pair<U1, U2> &other) : first(other.first), second(other.second) {}
  };
  
  typedef __typeof__(sizeof(int)) size_t;
  
  template<typename T>
  class vector {
    T *_start;
    T *_finish;
    T *_end_of_storage;
  public:
    vector() : _start(0), _finish(0), _end_of_storage(0) {}
    ~vector();
    
    size_t size() const {
      return size_t(_finish - _start);
    }
    
    void push_back();
    T pop_back();

    T &operator[](size_t n) {
      return _start[n];
    }
    
    const T &operator[](size_t n) const {
      return _start[n];
    }
    
    T *begin() { return _start; }
    const T *begin() const { return _start; }

    T *end() { return _finish; }
    const T *end() const { return _finish; }
  };
  
  class exception {
  public:
    exception() throw();
    virtual ~exception() throw();
    virtual const char *what() const throw() {
      return 0;
    }
  };
}
