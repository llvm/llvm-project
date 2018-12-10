#include <chrono>
#include <thread>

int main(int argc, char const *argv[])
{
    lldb_enable_attach();

    std::this_thread::sleep_for(std::chrono::seconds(30));
}
