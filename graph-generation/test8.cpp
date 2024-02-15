class A {
  public:
    int *data;

    A(int *val) : data(val) {}

    int getValue() const {
        return *data; // sink
    }
};

void modifyPointer(A *&ptr) {
    ptr->data = nullptr; // source
    return;
}

int useAlias(const A &alias) {
    int value = alias.getValue();
    return value;
}

int main() {
    int x = 42;
    A *p = new A(&x);

    modifyPointer(p);

    A *q = p;

    useAlias(*q);

    return 0;
}