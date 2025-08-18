# Overview

Arcade provides templates for public (`/templates`) and 1ES pipeline templates (`/templates-official`) scenarios.  Pipelines which are required to be managed by 1ES pipeline templates should reference `/templates-offical`, all other pipelines may reference `/templates`.

## How to use

Basic guidance is:

- 1ES Pipeline Template or 1ES Microbuild template runs should reference `eng/common/templates-official`. Any internal production-graded pipeline should use these templates.

- All other runs should reference `eng/common/templates`.

See [azure-pipelines.yml](../../azure-pipelines.yml) (templates-official example) or [azure-pipelines-pr.yml](../../azure-pipelines-pr.yml) (templates example) for examples.

#### The `templateIs1ESManaged` parameter

The `templateIs1ESManaged` is available on most templates and affects which of the variants is used for nested templates. See [Development Notes](#development-notes) below for more information on the `templateIs1ESManaged1 parameter.

- For templates under `job/`, `jobs/`, `steps`, or `post-build/`, this parameter must be explicitly set.

## Multiple outputs

1ES pipeline templates impose a policy where every publish artifact execution results in additional security scans being injected into your pipeline.  When using `templates-official/jobs/jobs.yml`, Arcade reduces the number of additional security injections by gathering all publishing outputs into the [Build.ArtifactStagingDirectory](https://learn.microsoft.com/en-us/azure/devops/pipelines/build/variables?view=azure-devops&tabs=yaml#build-variables-devops-services), and utilizing the [outputParentDirectory](https://eng.ms/docs/cloud-ai-platform/devdiv/one-engineering-system-1es/1es-docs/1es-pipeline-templates/features/outputs#multiple-outputs) feature of 1ES pipeline templates.  When implementing your pipeline, if you ensure publish artifacts are located in the `$(Build.ArtifactStagingDirectory)`, and utilize the 1ES provided template context, then you can reduce the number of security scans for your pipeline.

Example:
``` yaml
# azure-pipelines.yml
extends:
  template: azure-pipelines/MicroBuild.1ES.Official.yml@MicroBuildTemplate
  parameters:
    stages:
    - stage: build
      jobs:
      - template: /eng/common/templates-official/jobs/jobs.yml@self
        parameters:
          # 1ES makes use of outputs to reduce security task injection overhead
          templateContext:
            outputs:
            - output: pipelineArtifact
              displayName: 'Publish logs from source'
              continueOnError: true
              condition: always()
              targetPath: $(Build.ArtifactStagingDirectory)/artifacts/log
              artifactName: Logs
          jobs:
          - job: Windows
            steps:
            - script: echo "friendly neighborhood" > artifacts/marvel/spiderman.txt
          # copy build outputs to artifact staging directory for publishing
          - task: CopyFiles@2
              displayName: Gather build output
              inputs:
                SourceFolder: '$(System.DefaultWorkingDirectory)/artifacts/marvel'
                Contents: '**'
                TargetFolder: '$(Build.ArtifactStagingDirectory)/artifacts/marvel'
```

Note: Multiple outputs are ONLY applicable to 1ES PT publishing (only usable when referencing `templates-official`).

## Development notes

**Folder / file structure**

``` text
eng\common\
    [templates || templates-official]\
        job\
            job.yml                          (shim + artifact publishing logic)
            onelocbuild.yml                  (shim)
            publish-build-assets.yml         (shim)
            source-build.yml                 (shim)
            source-index-stage1.yml          (shim)
        jobs\
            codeql-build.yml                 (shim)
            jobs.yml                         (shim)
            source-build.yml                 (shim)
        post-build\
            post-build.yml                   (shim)
            common-variabls.yml              (shim)
            setup-maestro-vars.yml           (shim)
        steps\
            publish-build-artifacts.yml      (logic)
            publish-pipeline-artifacts.yml   (logic)
            component-governance.yml         (shim)
            generate-sbom.yml                (shim)
            publish-logs.yml                 (shim)
            retain-build.yml                 (shim)
            send-to-helix.yml                (shim)
            source-build.yml                 (shim)
        variables\
            pool-providers.yml               (logic + redirect) # templates/variables/pool-providers.yml will redirect to templates-official/variables/pool-providers.yml if you are running in the internal project
            sdl-variables.yml                (logic)
    core-templates\
        job\
            job.yml                          (logic)
            onelocbuild.yml                  (logic)
            publish-build-assets.yml         (logic)
            source-build.yml                 (logic)
            source-index-stage1.yml          (logic)
        jobs\
            codeql-build.yml                 (logic)
            jobs.yml                         (logic)
            source-build.yml                 (logic)
        post-build\
            common-variabls.yml              (logic)
            post-build.yml                   (logic)
            setup-maestro-vars.yml           (logic)
        steps\
            component-governance.yml         (logic)
            generate-sbom.yml                (logic)
            publish-build-artifacts.yml      (redirect)
            publish-logs.yml                 (logic)
            publish-pipeline-artifacts.yml   (redirect)
            retain-build.yml                 (logic)
            send-to-helix.yml                (logic)
            source-build.yml                 (logic)
        variables\
            pool-providers.yml               (redirect)
```

In the table above, a file is designated as "shim", "logic", or "redirect".

- shim - represents a yaml file which is an intermediate step between pipeline logic and .Net Core Engineering's templates (`core-templates`) and defines the `is1ESPipeline` parameter value.

- logic - represents actual base template logic.

- redirect- represents a file in `core-templates` which redirects to the "logic" file in either `templates` or `templates-official`.

Logic for Arcade's templates live **primarily** in the `core-templates` folder.  The exceptions to the location of the logic files are around artifact publishing, which is handled differently between 1es pipeline templates and standard templates.  `templates` and `templates-official` provide shim entry points which redirect to `core-templates` while also defining the `is1ESPipeline` parameter.  If a shim is referenced in `templates`, then `is1ESPipeline` is set to `false`.  If a shim is referenced in `templates-official`, then `is1ESPipeline` is set to `true`.

Within `templates` and `templates-official`, the templates at the "stages", and "jobs" / "job" level have been replaced with shims.  Templates at the "steps" and "variables" level are typically too granular to be replaced with shims and instead persist logic which is directly applicable to either scenario.

Within `core-templates`, there are a handful of places where logic is dependent on which shim entry point was used.  In those places, we redirect back to the respective logic file in `templates` or `templates-official`.
