#include <unistd.h>
#include <string.h>

#include <support/support.h>

struct statclass
{
  statclass()
  {
    write_message ("statclass\n");
  }
  ~statclass()
  {
    write_message ("~statclass\n");
  }
};

struct extclass
{
  ~extclass()
  {
    static statclass var;
  }
};

extclass globvar;
