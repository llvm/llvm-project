int main(int argc, char** argv) {
    int arg = 5;
    for (unsigned long long i = 0; i < 10'000'000'000; ++i) {
        arg *= 1;
    }
    return 0;
}