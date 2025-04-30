#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test int8toq  ########


qp55: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp55.f08 fcheck.$(OBJX)
	-$(RM) qp55.$(EXESUFFIX) qp55.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp55.f08 -o qp55.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp55.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp55.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp55
	qp55.$(EXESUFFIX)

verify: ;

qp55.run: run

