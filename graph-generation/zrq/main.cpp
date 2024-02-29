struct Node {
    int data;
};

int main(int argc, char **argv) {
    int x = 1024;
    Node *p = new Node();
    if (argc > 1) {
        p = nullptr;
    }
    if (x > 0 && p->data > 0) {
        p->data = 0;
    }
    return 0;
}