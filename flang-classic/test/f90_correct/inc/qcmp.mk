#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn08  ########


qcmp: run
	

build:  $(SRC)/qcmp.f08
	-$(RM) qcmp.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qcmp.f08 -o qcmp.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qcmp.$(OBJX) check.$(OBJX) $(LIBS) -o qcmp.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qcmp
	qcmp.$(EXESUFFIX)

verify: ;

qcmp.run: run

