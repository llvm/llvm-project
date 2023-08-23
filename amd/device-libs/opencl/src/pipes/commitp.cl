/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "pipes.h"

#define ATTR __attribute__((always_inline))

#define COMMIT_READ_PIPE_SIZE(SIZE, STYPE) \
ATTR void \
__commit_read_pipe_##SIZE(__global struct pipeimp* p, size_t rid) \
{ \
}

// DO_PIPE_SIZE(COMMIT_READ_PIPE_SIZE)

ATTR void
__commit_read_pipe(__global struct pipeimp* p, size_t rid, uint size, uint align)
{
}

#define COMMIT_WRITE_PIPE_SIZE(SIZE, STYPE) \
ATTR void \
__commit_write_pipe_##SIZE(__global struct pipeimp* p, size_t rid) \
{ \
}

// DO_PIPE_SIZE(COMMIT_WRITE_PIPE_SIZE)

ATTR void
__commit_write_pipe(__global struct pipeimp* p, size_t rid, uint size, uint align)
{
}

// Work group functions

#define WORK_GROUP_COMMIT_READ_PIPE_SIZE(SIZE, STYPE) \
ATTR void \
__work_group_commit_read_pipe_##SIZE(__global struct pipeimp* p, size_t rid) \
{ \
}

// DO_PIPE_SIZE(WORK_GROUP_COMMIT_READ_PIPE_SIZE)

ATTR void
__work_group_commit_read_pipe(__global struct pipeimp* p, size_t rid, uint size, uint align)
{
}

#define WORK_GROUP_COMMIT_WRITE_PIPE_SIZE(SIZE, STYPE) \
ATTR void \
__work_group_commit_write_pipe_##SIZE(__global struct pipeimp* p, size_t rid) \
{ \
}

// DO_PIPE_SIZE(WORK_GROUP_COMMIT_WRITE_PIPE_SIZE)

ATTR void
__work_group_commit_write_pipe(__global struct pipeimp* p, size_t rid, uint size, uint align)
{
}

// sub group functions

#define SUB_GROUP_COMMIT_READ_PIPE_SIZE(SIZE, STYPE) \
ATTR void \
__sub_group_commit_read_pipe_##SIZE(__global struct pipeimp* p, size_t rid) \
{ \
}

// DO_PIPE_SIZE(SUB_GROUP_COMMIT_READ_PIPE_SIZE)

ATTR void
__sub_group_commit_read_pipe(__global struct pipeimp* p, size_t rid, uint size, uint align)
{
}

#define SUB_GROUP_COMMIT_WRITE_PIPE_SIZE(SIZE, STYPE) \
ATTR void \
__sub_group_commit_write_pipe_##SIZE(__global struct pipeimp* p, size_t rid) \
{ \
}

// DO_PIPE_SIZE(SUB_GROUP_COMMIT_WRITE_PIPE_SIZE)

ATTR void
__sub_group_commit_write_pipe(__global struct pipeimp* p, size_t rid, uint size, uint align)
{
}

