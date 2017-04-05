import lit.formats
import lit.TestRunner

# Custom format class for static analyzer tests
class AnalyzerTest(lit.formats.ShTest):

    def execute(self, test, litConfig):
        result = self.executeWithAnalyzeSubstitution(
            test, litConfig, '-analyzer-constraints=range')

        if result.code == lit.Test.FAIL:
            return result

        # If z3 backend available, add an additional run line for it
        if test.config.clang_staticanalyzer_z3 == '1':
            result = self.executeWithAnalyzeSubstitution(
                test, litConfig, '-analyzer-constraints=z3 -DANALYZER_CM_Z3')

        return result

    def executeWithAnalyzeSubstitution(self, test, litConfig, substitution):
        saved_substitutions = list(test.config.substitutions)
        test.config.substitutions.append(('%analyze', substitution))
        result = lit.TestRunner.executeShTest(test, litConfig,
                                              self.execute_external)
        test.config.substitutions = saved_substitutions

        return result
