/*
 * Dell MIL-SPEC Event Monitor
 * Enhanced userspace monitoring utility for dell-milspec driver
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <getopt.h>
#include <sys/epoll.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <linux/dell-milspec.h>

#define MAX_EVENTS 10
#define BUF_SIZE 4096
#define SYSFS_BASE "/sys/devices/platform/dell-milspec"

/* Global flag for clean shutdown */
static volatile int running = 1;

/* Monitoring modes */
enum monitor_mode {
    MODE_SYSFS = 1,
    MODE_CHARDEV = 2,
    MODE_BOTH = 3
};

/* Configuration */
struct config {
    enum monitor_mode mode;
    int verbose;
    int show_timestamp;
    int use_colors;
    const char *device_path;
    const char *log_file;
    FILE *log_fp;
};

/* Color codes */
#define COLOR_RESET   "\033[0m"
#define COLOR_RED     "\033[31m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_BLUE    "\033[34m"
#define COLOR_MAGENTA "\033[35m"
#define COLOR_CYAN    "\033[36m"

/* Event type strings */
static const char *event_type_str[] = {
    [MILSPEC_EVENT_BOOT] = "BOOT",
    [MILSPEC_EVENT_ACTIVATION] = "ACTIVATION",
    [MILSPEC_EVENT_MODE_CHANGE] = "MODE_CHANGE",
    [MILSPEC_EVENT_ERROR] = "ERROR",
    [MILSPEC_EVENT_USER_REQUEST] = "USER_REQUEST",
    [MILSPEC_EVENT_SECURITY] = "SECURITY",
    [MILSPEC_EVENT_POWER] = "POWER",
    [MILSPEC_EVENT_FIRMWARE] = "FIRMWARE",
    [MILSPEC_EVENT_INTRUSION] = "INTRUSION",
    [MILSPEC_EVENT_CRYPTO] = "CRYPTO",
};

/* Mode 5 level strings */
static const char *mode5_level_str[] = {
    "DISABLED",
    "STANDARD",
    "ENHANCED",
    "PARANOID",
    "PARANOID_PLUS"
};

/* Signal handler for clean shutdown */
static void signal_handler(int sig)
{
    running = 0;
}

/* Get timestamp string */
static void get_timestamp(char *buf, size_t size)
{
    struct timespec ts;
    struct tm tm;

    clock_gettime(CLOCK_REALTIME, &ts);
    localtime_r(&ts.tv_sec, &tm);

    snprintf(buf, size, "%04d-%02d-%02d %02d:%02d:%02d.%03ld",
             tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
             tm.tm_hour, tm.tm_min, tm.tm_sec, ts.tv_nsec / 1000000);
}

/* Print with optional color */
static void print_color(struct config *cfg, const char *color, const char *fmt, ...)
{
    va_list args;

    if (cfg->use_colors && isatty(STDOUT_FILENO))
        printf("%s", color);

    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);

    if (cfg->use_colors && isatty(STDOUT_FILENO))
        printf("%s", COLOR_RESET);

    /* Also log to file if configured */
    if (cfg->log_fp) {
        va_start(args, fmt);
        vfprintf(cfg->log_fp, fmt, args);
        va_end(args);
        fflush(cfg->log_fp);
    }
}

/* Format and print event */
static void print_event(struct config *cfg, struct milspec_event *event)
{
    char timestamp[64];
    const char *type_str = "UNKNOWN";
    const char *color = COLOR_RESET;

    if (event->event_type < sizeof(event_type_str) / sizeof(event_type_str[0]))
        type_str = event_type_str[event->event_type];

    /* Choose color based on event type */
    switch (event->event_type) {
    case MILSPEC_EVENT_ERROR:
        color = COLOR_RED;
        break;
    case MILSPEC_EVENT_SECURITY:
    case MILSPEC_EVENT_INTRUSION:
        color = COLOR_MAGENTA;
        break;
    case MILSPEC_EVENT_ACTIVATION:
        color = COLOR_GREEN;
        break;
    case MILSPEC_EVENT_MODE_CHANGE:
        color = COLOR_YELLOW;
        break;
    default:
        color = COLOR_CYAN;
    }

    if (cfg->show_timestamp) {
        get_timestamp(timestamp, sizeof(timestamp));
        printf("[%s] ", timestamp);
    }

    print_color(cfg, color, "%-12s", type_str);
    printf(" | Data: [%u, %u] | %s\n", event->data1, event->data2, event->message);
}

/* Monitor character device for events */
static int monitor_chardev(struct config *cfg, int fd)
{
    struct milspec_events events;
    int ret;

    ret = ioctl(fd, MILSPEC_IOC_GET_EVENTS, &events);
    if (ret < 0) {
        perror("ioctl(GET_EVENTS)");
        return -1;
    }

    if (events.count > 0) {
        if (cfg->verbose)
            printf("Received %u events (lost: %u)\n", events.count, events.lost);

        for (unsigned int i = 0; i < events.count; i++) {
            print_event(cfg, &events.events[i]);
        }
    }

    return events.count;
}

/* Monitor sysfs files */
static int setup_sysfs_monitoring(int epfd, struct config *cfg)
{
    static const char *sysfs_files[] = {
        "activation_log",
        "mode5",
        "dsmil",
        "crypto_status",
        NULL
    };

    for (int i = 0; sysfs_files[i]; i++) {
        char path[256];
        int fd;
        struct epoll_event ev;

        snprintf(path, sizeof(path), "%s/%s", SYSFS_BASE, sysfs_files[i]);

        fd = open(path, O_RDONLY);
        if (fd < 0) {
            if (cfg->verbose)
                fprintf(stderr, "Warning: Cannot open %s: %s\n", path, strerror(errno));
            continue;
        }

        ev.events = EPOLLPRI | EPOLLERR;
        ev.data.fd = fd;

        if (epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev) < 0) {
            perror("epoll_ctl");
            close(fd);
            continue;
        }

        if (cfg->verbose)
            printf("Monitoring %s\n", path);
    }

    return 0;
}

/* Print current status */
static void print_status(struct config *cfg, int fd)
{
    struct milspec_status status;
    struct milspec_security security;

    if (ioctl(fd, MILSPEC_IOC_GET_STATUS, &status) == 0) {
        print_color(cfg, COLOR_GREEN, "\n=== MIL-SPEC Status ===\n");
        printf("API Version: 0x%06x\n", status.api_version);
        printf("Mode 5: %s (Level: %s)\n",
               status.mode5_enabled ? "ENABLED" : "DISABLED",
               mode5_level_str[status.mode5_level]);
        printf("DSMIL Mode: %d\n", status.dsmil_mode);
        printf("Service Mode: %s\n", status.service_mode ? "ACTIVE" : "INACTIVE");
        printf("Boot Progress: 0x%02x\n", status.boot_progress);
        printf("Activation Count: %u\n", status.activation_count);
        printf("Error Count: %u\n", status.error_count);

        /* Show active DSMIL devices */
        printf("Active DSMIL Devices:");
        for (int i = 0; i < 10; i++) {
            if (status.dsmil_active[i])
                printf(" DSMIL0D%d", i);
        }
        printf("\n");
    }

    if (ioctl(fd, MILSPEC_IOC_GET_SECURITY, &security) == 0) {
        print_color(cfg, COLOR_YELLOW, "\n=== Security Status ===\n");
        printf("Intrusion Detected: %s\n", security.intrusion_detected ? "YES!" : "No");
        printf("Tamper Detected: %s\n", security.tamper_detected ? "YES!" : "No");
        printf("Secure Boot: %s\n", security.secure_boot_enabled ? "Enabled" : "Disabled");
        printf("Measured Boot: %s\n", security.measured_boot ? "Yes" : "No");
        printf("Encrypted Memory: %s\n", security.encrypted_memory ? "Active" : "Inactive");
        printf("DMA Protection: %s\n", security.dma_protection ? "Enabled" : "Disabled");

        if (security.crypto_serial[0]) {
            printf("Crypto Chip Serial: ");
            for (int i = 0; i < 9; i++)
                printf("%02X", security.crypto_serial[i]);
            printf("\n");
        }
    }
}

/* Print usage */
static void usage(const char *prog)
{
    printf("Usage: %s [options]\n", prog);
    printf("\nOptions:\n");
    printf("  -d, --device PATH    Character device path (default: /dev/milspec)\n");
    printf("  -m, --mode MODE      Monitor mode: sysfs, chardev, both (default: both)\n");
    printf("  -s, --status         Show current status and exit\n");
    printf("  -t, --timestamp      Show timestamps\n");
    printf("  -c, --color          Use color output\n");
    printf("  -l, --log FILE       Log events to file\n");
    printf("  -v, --verbose        Verbose output\n");
    printf("  -h, --help           Show this help\n");
    printf("\nExamples:\n");
    printf("  %s -tc                    # Monitor with timestamps and color\n", prog);
    printf("  %s -m chardev -l log.txt  # Monitor char device, log to file\n", prog);
    printf("  %s -s                     # Show status and exit\n", prog);
}

int main(int argc, char *argv[])
{
    struct config cfg = {
        .mode = MODE_BOTH,
        .device_path = "/dev/milspec",
        .show_timestamp = 0,
        .use_colors = 0,
        .verbose = 0,
        .log_file = NULL,
        .log_fp = NULL
    };

    int opt, option_index = 0;
    int epfd = -1, chardev_fd = -1;
    int show_status_only = 0;
    struct epoll_event events[MAX_EVENTS];

    static struct option long_options[] = {
        {"device", required_argument, 0, 'd'},
        {"mode", required_argument, 0, 'm'},
        {"status", no_argument, 0, 's'},
        {"timestamp", no_argument, 0, 't'},
        {"color", no_argument, 0, 'c'},
        {"log", required_argument, 0, 'l'},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    /* Parse command line */
    while ((opt = getopt_long(argc, argv, "d:m:stcl:vh", long_options, &option_index)) != -1) {
        switch (opt) {
        case 'd':
            cfg.device_path = optarg;
            break;
        case 'm':
            if (strcmp(optarg, "sysfs") == 0)
                cfg.mode = MODE_SYSFS;
            else if (strcmp(optarg, "chardev") == 0)
                cfg.mode = MODE_CHARDEV;
            else if (strcmp(optarg, "both") == 0)
                cfg.mode = MODE_BOTH;
            else {
                fprintf(stderr, "Invalid mode: %s\n", optarg);
                exit(1);
            }
            break;
        case 's':
            show_status_only = 1;
            break;
        case 't':
            cfg.show_timestamp = 1;
            break;
        case 'c':
            cfg.use_colors = 1;
            break;
        case 'l':
            cfg.log_file = optarg;
            break;
        case 'v':
            cfg.verbose = 1;
            break;
        case 'h':
            usage(argv[0]);
            exit(0);
        default:
            usage(argv[0]);
            exit(1);
        }
    }

    /* Open log file if specified */
    if (cfg.log_file) {
        cfg.log_fp = fopen(cfg.log_file, "a");
        if (!cfg.log_fp) {
            perror("fopen log file");
            exit(1);
        }
    }

    /* Open character device if needed */
    if (cfg.mode & MODE_CHARDEV || show_status_only) {
        chardev_fd = open(cfg.device_path, O_RDWR);
        if (chardev_fd < 0) {
            perror("open character device");
            exit(1);
        }
    }

    /* Show status and exit if requested */
    if (show_status_only) {
        print_status(&cfg, chardev_fd);
        close(chardev_fd);
        exit(0);
    }

    /* Set up signal handlers */
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    /* Create epoll instance */
    epfd = epoll_create1(0);
    if (epfd < 0) {
        perror("epoll_create1");
        exit(1);
    }

    /* Set up sysfs monitoring if requested */
    if (cfg.mode & MODE_SYSFS) {
        setup_sysfs_monitoring(epfd, &cfg);
    }

    /* Add character device to epoll if monitoring it */
    if (cfg.mode & MODE_CHARDEV) {
        struct epoll_event ev;
        ev.events = EPOLLIN;
        ev.data.fd = chardev_fd;

        if (epoll_ctl(epfd, EPOLL_CTL_ADD, chardev_fd, &ev) < 0) {
            perror("epoll_ctl chardev");
            exit(1);
        }
    }

    print_color(&cfg, COLOR_GREEN, "=== MIL-SPEC Event Monitor Started ===\n");
    printf("Monitoring mode: ");
    switch (cfg.mode) {
    case MODE_SYSFS:
        printf("sysfs only\n");
        break;
    case MODE_CHARDEV:
        printf("character device only\n");
        break;
    case MODE_BOTH:
        printf("both sysfs and character device\n");
        break;
    }

    if (cfg.log_file)
        printf("Logging to: %s\n", cfg.log_file);

    printf("Press Ctrl+C to exit\n\n");

    /* Main event loop */
    while (running) {
        int n = epoll_wait(epfd, events, MAX_EVENTS, 1000);

        if (n < 0) {
            if (errno == EINTR)
                continue;
            perror("epoll_wait");
            break;
        }

        /* Handle events */
        for (int i = 0; i < n; i++) {
            if (events[i].data.fd == chardev_fd) {
                /* Character device event */
                monitor_chardev(&cfg, chardev_fd);
            } else {
                /* Sysfs file event */
                char buf[BUF_SIZE];
                int fd = events[i].data.fd;

                lseek(fd, 0, SEEK_SET);
                ssize_t len = read(fd, buf, sizeof(buf) - 1);

                if (len > 0) {
                    buf[len] = '\0';
                    print_color(&cfg, COLOR_BLUE, "[SYSFS] ");
                    printf("%s", buf);
                }
            }
        }

        /* Poll character device periodically if monitoring it */
        if ((cfg.mode & MODE_CHARDEV) && n == 0) {
            monitor_chardev(&cfg, chardev_fd);
        }
    }

    print_color(&cfg, COLOR_YELLOW, "\n=== MIL-SPEC Event Monitor Stopped ===\n");

    /* Cleanup */
    if (epfd >= 0)
        close(epfd);
    if (chardev_fd >= 0)
        close(chardev_fd);
    if (cfg.log_fp)
        fclose(cfg.log_fp);

    return 0;
}
