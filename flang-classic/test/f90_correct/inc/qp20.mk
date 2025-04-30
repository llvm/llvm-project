#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp20  ########


qp20: run


build:  $(SRC)/qp20.f08
	-$(RM) qp20.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp20.f08 -o qp20.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp20.$(OBJX) check.$(OBJX) $(LIBS) -o qp20.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp20
	qp20.$(EXESUFFIX)

verify: ;
