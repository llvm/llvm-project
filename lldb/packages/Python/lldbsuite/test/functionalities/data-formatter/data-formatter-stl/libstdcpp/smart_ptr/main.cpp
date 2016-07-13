#include <memory>
#include <string>

int
main()
{
    std::shared_ptr<char> nsp;
    std::shared_ptr<int> isp(new int{123});
    std::shared_ptr<std::string> ssp = std::make_shared<std::string>("foobar");

    std::weak_ptr<char> nwp;
    std::weak_ptr<int> iwp = isp;
    std::weak_ptr<std::string> swp = ssp;

    nsp.reset(); // Set break point at this line.
    isp.reset();
    ssp.reset();

    return 0; // Set break point at this line.
}
