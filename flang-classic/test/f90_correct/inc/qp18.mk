#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp18  ########


qp18: run


build:  $(SRC)/qp18.f08
	-$(RM) qp18.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp18.f08 -o qp18.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp18.$(OBJX) check.$(OBJX) $(LIBS) -o qp18.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp18
	qp18.$(EXESUFFIX)

verify: ;
