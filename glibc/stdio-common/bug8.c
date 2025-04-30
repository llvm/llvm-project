#include <stdio.h>
#include <string.h>

int
main(int argc, char *argv[])
{
    char buf[100];
    int point, x, y;
    int status = 0;

    sscanf("0x10 10", "%x %x", &x, &y);
    sprintf(buf, "%d %d", x, y);
    puts (buf);
    status |= strcmp (buf, "16 16");
    sscanf("P012349876", "P%1d%4d%4d", &point, &x, &y);
    sprintf(buf, "%d %d %d", point, x, y);
    status |= strcmp (buf, "0 1234 9876");
    puts (buf);
    sscanf("P112349876", "P%1d%4d%4d", &point, &x, &y);
    sprintf(buf, "%d %d %d", point, x, y);
    status |= strcmp (buf, "1 1234 9876");
    puts (buf);

    puts (status ? "Test failed" : "Test passed");

    return status;
}
