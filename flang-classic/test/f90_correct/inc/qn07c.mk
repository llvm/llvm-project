#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn07c  ########


qn07c: run
	

build:  $(SRC)/qn07c.f90
	-$(RM) qn07c.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn07c.f90 -o qn07c.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn07c.$(OBJX) check.$(OBJX) $(LIBS) -o qn07c.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn07c
	qn07c.$(EXESUFFIX)

verify: ;

qn07c.run: run

