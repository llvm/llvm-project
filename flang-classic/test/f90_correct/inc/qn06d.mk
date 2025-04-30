#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn06d  ########


qn06d: run
	

build:  $(SRC)/qn06d.f90
	-$(RM) qn06d.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qn06d.f90 -o qn06d.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qn06d.$(OBJX) check.$(OBJX) $(LIBS) -o qn06d.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn06d
	qn06d.$(EXESUFFIX)

verify: ;

qn06d.run: run

