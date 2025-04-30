#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp12  ########


qp12: run


build:  $(SRC)/qp12.f08
	-$(RM) qp12.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp12.f08 -o qp12.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp12.$(OBJX) check.$(OBJX) $(LIBS) -o qp12.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp12
	qp12.$(EXESUFFIX)

verify: ;
