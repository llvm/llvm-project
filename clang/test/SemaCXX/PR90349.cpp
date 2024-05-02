// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only %s

// expected-no-diagnostics

namespace std {
template<class T>
concept floating_point = __is_same(T,double) || __is_same(T,float);

template<class T>
concept integral = __is_same(T,int);

}

template<std::integral T, std::floating_point Float>
class Blob;

template<std::floating_point Float, std::integral T>
Blob<T, Float> MakeBlob();

template<std::integral T, std::floating_point Float>
class Blob {
private:
    Blob() {}

    friend Blob<T, Float> MakeBlob<Float, T>();
};

template<std::floating_point Float, std::integral T>
Blob<T, Float> MakeBlob()
{
    return Blob<T, Float>();
}

template<std::floating_point Float, std::integral T>
Blob<T, Float> FindBlobs()
{
    return MakeBlob<Float, T>();
}

int main(int argc, const char * argv[]) {
    FindBlobs<double, int>();
    return 0;
}
