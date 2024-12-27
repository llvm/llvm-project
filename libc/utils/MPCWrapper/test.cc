#include<iostream>
using namespace std;

class shogo {
public:
    void Run();
    void Hey();
};

void shogo::Run() {
    cout << "This is Run" << endl;
}

int main() {
    shogo s;
    s.Run();
}
