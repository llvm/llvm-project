// RUN: %clangxx -O0 %s -o %t && %run %t

#include <assert.h>
#include <sys/types.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
	pid_t pid;
	ptrace_lwpinfo pl;

	if ((pid = fork()) == 0) {
    		execl("/bin/true", "true", NULL);
		exit(-1);
	}
	assert(pid != -1);
	waitpid(pid, NULL, 0);
	auto r = ptrace(PT_ATTACH, pid, NULL, 0);
	assert(!r);
	waitpid(pid, NULL, 0);
	r = ptrace(PT_LWPINFO, pid, reinterpret_cast<caddr_t>(&pl), sizeof(pl));
	assert(!r);
	r = ptrace(PT_DETACH, pid, NULL, 0);
	return 0;
}
