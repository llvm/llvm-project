#define ARGS 1, 2, 3, 4
#define CALL(x) foo(10, 11, 12, x)

#define NAME p
#define FIELD data

struct Node {
    int data;
};

void foo(int a, int b, int c, int d) {
    // ...
}

int main() {
    foo(5, 6, 7, 8);
    foo(ARGS);
    CALL(13);

    Node *p = new Node();
    p->FIELD;
    NAME->data;
    NAME->FIELD;

    return 0;
}