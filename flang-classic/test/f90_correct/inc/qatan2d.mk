#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test atan2  ########


qatan2d: run


build:  $(SRC)/qatan2d.f08
	-$(RM) qatan2d.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qatan2d.f08 -o qatan2d.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qatan2d.$(OBJX) check.$(OBJX) $(LIBS) -o qatan2d.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qatan2d
	qatan2d.$(EXESUFFIX)

verify: ;
