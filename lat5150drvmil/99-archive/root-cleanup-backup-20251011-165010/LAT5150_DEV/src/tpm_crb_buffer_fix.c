#include <stdio.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

/* TPM CRB Register Offsets */
#define TPM_CRB_BASE_ADDR       0xFED40000
#define TPM_CRB_CTRL_REQ        0x40
#define TPM_CRB_CTRL_STS        0x44
#define TPM_CRB_CTRL_CANCEL     0x48
#define TPM_CRB_CTRL_START      0x4C
#define TPM_CRB_INT_ENABLE      0x50
#define TPM_CRB_INT_STS         0x54
#define TPM_CRB_CMD_SIZE        0x58
#define TPM_CRB_CMD_ADDR_L      0x5C
#define TPM_CRB_CMD_ADDR_H      0x60
#define TPM_CRB_RSP_SIZE        0x64
#define TPM_CRB_RSP_ADDR_L      0x68
#define TPM_CRB_RSP_ADDR_H      0x6C

/* Buffer Fix Strategy */
#define ALIGNED_BUFFER_SIZE     0x1000  /* 4KB aligned */

int fix_crb_buffers(void) {
    int fd;
    volatile uint32_t *crb_regs;
    uint32_t cmd_size, rsp_size;
    uint64_t cmd_addr, rsp_addr;

    fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("Cannot open /dev/mem");
        return -1;
    }

    /* Map TPM CRB registers */
    crb_regs = mmap(NULL, 0x1000, PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd, TPM_CRB_BASE_ADDR);
    if (crb_regs == MAP_FAILED) {
        perror("mmap failed");
        close(fd);
        return -1;
    }

    /* Read current buffer configuration */
    cmd_size = crb_regs[TPM_CRB_CMD_SIZE/4];
    rsp_size = crb_regs[TPM_CRB_RSP_SIZE/4];
    cmd_addr = ((uint64_t)crb_regs[TPM_CRB_CMD_ADDR_H/4] << 32) |
               crb_regs[TPM_CRB_CMD_ADDR_L/4];
    rsp_addr = ((uint64_t)crb_regs[TPM_CRB_RSP_ADDR_H/4] << 32) |
               crb_regs[TPM_CRB_RSP_ADDR_L/4];

    printf("Current CMD buffer: addr=0x%lx, size=0x%x\n", cmd_addr, cmd_size);
    printf("Current RSP buffer: addr=0x%lx, size=0x%x\n", rsp_addr, rsp_size);

    /* NSA Buffer Alignment Fix */
    if (cmd_size != rsp_size || cmd_addr == rsp_addr) {
        printf("Applying NSA buffer alignment fix...\n");

        /* Force non-overlapping aligned buffers */
        crb_regs[TPM_CRB_CMD_SIZE/4] = ALIGNED_BUFFER_SIZE;
        crb_regs[TPM_CRB_RSP_SIZE/4] = ALIGNED_BUFFER_SIZE;

        /* Ensure buffers don't overlap */
        if (cmd_addr == rsp_addr) {
            crb_regs[TPM_CRB_RSP_ADDR_L/4] = (uint32_t)(cmd_addr + ALIGNED_BUFFER_SIZE);
            crb_regs[TPM_CRB_RSP_ADDR_H/4] = (uint32_t)((cmd_addr + ALIGNED_BUFFER_SIZE) >> 32);
        }

        printf("Buffer fix applied successfully\n");
    }

    munmap((void*)crb_regs, 0x1000);
    close(fd);
    return 0;
}

int main(int argc, char **argv) {
    printf("NSA TPM CRB Buffer Alignment Tool\n");
    printf("Target: STMicroelectronics TPM0176\n\n");

    if (geteuid() != 0) {
        printf("Root access required for /dev/mem access\n");
        return 1;
    }

    return fix_crb_buffers();
}