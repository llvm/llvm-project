#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp04  ########


qp04: run


build:  $(SRC)/qp04.f08
	-$(RM) qp04.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp04.f08 -o qp04.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp04.$(OBJX) check.$(OBJX) $(LIBS) -o qp04.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp04
	qp04.$(EXESUFFIX)

verify: ;
