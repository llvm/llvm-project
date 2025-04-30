#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn07b  ########


qn07b: run
	

build:  $(SRC)/qn07b.f90
	-$(RM) qn07b.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn07b.f90 -o qn07b.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn07b.$(OBJX) check.$(OBJX) $(LIBS) -o qn07b.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn07b
	qn07b.$(EXESUFFIX)

verify: ;

qn07b.run: run

