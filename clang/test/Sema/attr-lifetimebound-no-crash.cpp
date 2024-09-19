// RUN: %clang_cc1 %s -verify -fsyntax-only

// expected-no-diagnostics

template<typename T>
struct Bar {
    int* data;

    auto operator[](const int index) const [[clang::lifetimebound]] -> decltype(data[index]) {
        return data[index];
    }
};

int main() {
    Bar<int> b;
    (void)b[2];
}
