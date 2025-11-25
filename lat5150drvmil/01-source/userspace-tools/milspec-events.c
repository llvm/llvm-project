#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/epoll.h>

int main(void)
{
    int fd, epfd;
    struct epoll_event ev, events[10];
    char buf[256];
    
    /* Open sysfs attribute that changes on events */
    fd = open("/sys/devices/platform/dell-milspec/activation_log", O_RDONLY);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    
    epfd = epoll_create1(0);
    ev.events = EPOLLPRI | EPOLLERR;
    ev.data.fd = fd;
    epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev);
    
    printf("Monitoring MIL-SPEC events...\n");
    
    while (1) {
        int n = epoll_wait(epfd, events, 10, -1);
        
        if (n > 0) {
            lseek(fd, 0, SEEK_SET);
            read(fd, buf, sizeof(buf));
            printf("Event: %s", buf);
        }
    }
    
    return 0;
}
