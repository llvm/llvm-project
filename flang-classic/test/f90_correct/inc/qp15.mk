#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp15  ########


qp15: run


build:  $(SRC)/qp15.f08
	-$(RM) qp15.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp15.f08 -o qp15.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp15.$(OBJX) check.$(OBJX) $(LIBS) -o qp15.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp15
	qp15.$(EXESUFFIX)

verify: ;
