#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn07d  ########


qn07d: run
	

build:  $(SRC)/qn07d.f90
	-$(RM) qn07d.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn07d.f90 -o qn07d.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn07d.$(OBJX) check.$(OBJX) $(LIBS) -o qn07d.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn07d
	qn07d.$(EXESUFFIX)

verify: ;

qn07d.run: run

