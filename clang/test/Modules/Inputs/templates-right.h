@__experimental_modules_import templates_top;

template<typename T> class Vector { 
public:
  void push_back(const T&);
};

template<typename T> class List;
template<> class List<bool> {
public:
  void push_back(int);
};

namespace N {
  template<typename T> class Set {
  public:
    void insert(T);
  };
}

template <typename T>
void pendingInstantiationEmit(T) {}
void triggerPendingInstantiationToo() {
  pendingInstantiationEmit(12);
}

void redeclDefinitionEmit(){}
