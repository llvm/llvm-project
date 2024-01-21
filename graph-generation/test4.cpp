int loop(int n) {
    int s = 0, x = 1;
    for (int i = 0; i < n; i++) {
        s = s + i;
        if (i < 0) {
            x *= i;
            break;
        } else if (i > 10) {
            continue;
        } else if (i > 20) {
            x *= 2 * i;
        } else {
            x = x / 2 * i;
        }
        s += i;
    }
    s += x;
    return s;
}

void foo() {
    int x = 0;
    x = x + 10;
    int ret = -1;
    ret = loop(x);
}