#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp10  ########


qp10: run


build:  $(SRC)/qp10.f08
	-$(RM) qp10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp10.f08 -o qp10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp10.$(OBJX) check.$(OBJX) $(LIBS) -o qp10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp10
	qp10.$(EXESUFFIX)

verify: ;
