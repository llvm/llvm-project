/*
 * DSMIL Driver Test Program
 * =========================
 *
 * Example userspace program demonstrating how to interact with the
 * DSMIL kernel driver (simplified educational version).
 *
 * Compile: gcc -o test-dsmil test-dsmil.c
 * Run: ./test-dsmil
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>

/* ioctl commands (must match kernel driver) */
#define DSMIL_IOCTL_READ  0x1001
#define DSMIL_IOCTL_WRITE 0x1002
#define DSMIL_IOCTL_COUNT 0x1003

void print_usage(const char *prog)
{
	printf("Usage: %s [command]\n", prog);
	printf("Commands:\n");
	printf("  read            - Read device status\n");
	printf("  write <msg>     - Write message to device\n");
	printf("  ioctl-read <id> - Simulate device read via ioctl\n");
	printf("  ioctl-write <id> <val> - Simulate device write via ioctl\n");
	printf("  count           - Get device count via ioctl\n");
	printf("  sysfs           - Show sysfs attributes\n");
	printf("  demo            - Run full demonstration\n");
}

int test_read(int fd)
{
	char buf[256];
	ssize_t bytes;

	printf("=== Testing read() ===\n");

	bytes = read(fd, buf, sizeof(buf) - 1);
	if (bytes < 0) {
		perror("read failed");
		return -1;
	}

	buf[bytes] = '\0';
	printf("Read %zd bytes:\n%s\n", bytes, buf);

	return 0;
}

int test_write(int fd, const char *msg)
{
	ssize_t bytes;

	printf("=== Testing write() ===\n");
	printf("Writing: %s\n", msg);

	bytes = write(fd, msg, strlen(msg));
	if (bytes < 0) {
		perror("write failed");
		return -1;
	}

	printf("Wrote %zd bytes\n", bytes);

	return 0;
}

int test_ioctl_read(int fd, unsigned int device_id)
{
	unsigned int value;
	int ret;

	printf("=== Testing ioctl read (device 0x%04X) ===\n", device_id);

	/* Note: This ioctl interface is simplified for learning */
	ret = ioctl(fd, DSMIL_IOCTL_READ, device_id);
	if (ret < 0) {
		perror("ioctl read failed");
		return -1;
	}

	printf("Device 0x%04X simulated value: 0x%08X\n", device_id, ret);

	return 0;
}

int test_ioctl_count(int fd)
{
	unsigned int count;
	int ret;

	printf("=== Testing ioctl get count ===\n");

	ret = ioctl(fd, DSMIL_IOCTL_COUNT, &count);
	if (ret < 0) {
		perror("ioctl count failed");
		return -1;
	}

	printf("Device count: %u\n", count);

	return 0;
}

void show_sysfs_attrs(void)
{
	char buf[256];
	FILE *fp;

	printf("=== Sysfs Attributes ===\n");

	/* Read device_count */
	fp = fopen("/sys/class/misc/dsmil/device_count", "r");
	if (fp) {
		if (fgets(buf, sizeof(buf), fp)) {
			printf("device_count: %s", buf);
		}
		fclose(fp);
	}

	/* Read access_count */
	fp = fopen("/sys/class/misc/dsmil/access_count", "r");
	if (fp) {
		if (fgets(buf, sizeof(buf), fp)) {
			printf("access_count: %s", buf);
		}
		fclose(fp);
	}

	/* Read version */
	fp = fopen("/sys/class/misc/dsmil/version", "r");
	if (fp) {
		if (fgets(buf, sizeof(buf), fp)) {
			printf("version: %s", buf);
		}
		fclose(fp);
	}

	/* Read status */
	fp = fopen("/sys/class/misc/dsmil/status", "r");
	if (fp) {
		if (fgets(buf, sizeof(buf), fp)) {
			printf("status: %s", buf);
		}
		fclose(fp);
	}

	printf("\n");
}

void run_demo(int fd)
{
	printf("\n");
	printf("╔══════════════════════════════════════════════════╗\n");
	printf("║   DSMIL Educational Driver Demonstration         ║\n");
	printf("╚══════════════════════════════════════════════════╝\n");
	printf("\n");

	/* Read device status */
	test_read(fd);
	printf("\n");

	/* Write a command */
	test_write(fd, "Hello DSMIL!\n");
	printf("\n");

	/* Test ioctl - read some devices */
	test_ioctl_read(fd, 0x8000);
	printf("\n");

	test_ioctl_read(fd, 0x8010);
	printf("\n");

	/* Get device count */
	test_ioctl_count(fd);
	printf("\n");

	/* Show sysfs attributes */
	show_sysfs_attrs();

	/* Reset access counter */
	printf("=== Resetting access counter ===\n");
	test_write(fd, "reset\n");
	printf("\n");

	/* Show sysfs again to see reset */
	show_sysfs_attrs();

	printf("╔══════════════════════════════════════════════════╗\n");
	printf("║   Demonstration Complete!                        ║\n");
	printf("║                                                  ║\n");
	printf("║   Check kernel log: sudo dmesg | grep dsmil-sim ║\n");
	printf("╚══════════════════════════════════════════════════╝\n");
	printf("\n");
}

int main(int argc, char **argv)
{
	int fd;
	int ret = 0;

	printf("DSMIL Driver Test Program\n");
	printf("=========================\n\n");

	/* Open the device */
	fd = open("/dev/dsmil", O_RDWR);
	if (fd < 0) {
		perror("Failed to open /dev/dsmil");
		printf("\nMake sure the kernel module is loaded:\n");
		printf("  sudo insmod dsmil-simple.ko\n");
		printf("  or\n");
		printf("  sudo modprobe dsmil-simple\n");
		return 1;
	}

	printf("Device opened successfully: /dev/dsmil\n\n");

	/* Parse command */
	if (argc < 2 || strcmp(argv[1], "demo") == 0) {
		/* Run full demonstration */
		run_demo(fd);
	} else if (strcmp(argv[1], "read") == 0) {
		ret = test_read(fd);
	} else if (strcmp(argv[1], "write") == 0) {
		if (argc < 3) {
			printf("Error: write command requires a message\n");
			ret = -1;
		} else {
			ret = test_write(fd, argv[2]);
		}
	} else if (strcmp(argv[1], "ioctl-read") == 0) {
		if (argc < 3) {
			printf("Error: ioctl-read requires device ID\n");
			ret = -1;
		} else {
			unsigned int id = strtoul(argv[2], NULL, 16);
			ret = test_ioctl_read(fd, id);
		}
	} else if (strcmp(argv[1], "count") == 0) {
		ret = test_ioctl_count(fd);
	} else if (strcmp(argv[1], "sysfs") == 0) {
		show_sysfs_attrs();
	} else {
		print_usage(argv[0]);
		ret = -1;
	}

	/* Close the device */
	close(fd);

	return ret ? 1 : 0;
}
