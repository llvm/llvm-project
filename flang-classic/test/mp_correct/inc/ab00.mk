#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
ab00: ab00.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN2) ./a.$(EXESUFFIX) $(LOG)
ab00.$(OBJX): $(SRC)/ab00.F check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/ab00.F
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) ab00.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: ab00
run: ;
